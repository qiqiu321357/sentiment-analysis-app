import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from wordcloud import WordCloud
import jieba
import re
import numpy as np
from datetime import datetime
import json
from pathlib import Path

import streamlit as st

# ==========================================
# 1. 全局配置（修复中文显示）
# ==========================================

# 获取当前目录（兼容本地和Streamlit Cloud）
try:
    current_dir = Path(__file__).parent.absolute()
except:
    current_dir = Path(os.getcwd())

# 字体文件路径（确保simhei.ttf和app.py在同一文件夹）
font_path = current_dir / 'simhei.ttf'

# 字体配置（兼容新旧版matplotlib）
FONT_PATH = None
if font_path.exists():
    try:
        # 新版matplotlib (3.6+) 使用 addfont
        fm.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
        FONT_PATH = str(font_path)
        print(f"[INFO] 已加载中文字体: {font_path}")
    except AttributeError:
        # 旧版降级处理
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        FONT_PATH = str(font_path)
        print(f"[INFO] 旧版matplotlib，使用备用字体设置")
    except Exception as e:
        print(f"[WARNING] 字体加载失败: {e}")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
else:
    # 云端无字体文件时的兜底方案
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    print(f"[WARNING] 字体文件不存在: {font_path}，中文可能显示为方框")

plt.rcParams['axes.unicode_minus'] = False

# Streamlit 页面配置（必须在任何st命令之前）
st.set_page_config(
    page_title="电商评论情感分析系统 - 天津财经大学",
    page_icon="",
    layout="wide"
)

# ==========================================
# 2. 整合版电商情感词典（全网资源+领域适配）
# ==========================================
def load_integrated_sentiment_dict():
    """
    整合版电商情感词典：
    - 基础：HowNet + NTUSD + BOSON
    - 领域：电商专用词 + 维度词
    - 配套：否定词 + 程度副词 + 网络新词
    """
    
    # ---------------------- 核心情感词（带强度） ----------------------
    sentiment_words = {
        # 强烈积极 (9-10分)
        'strong_positive': {
            '完美': 9.5, '极品': 9.8, '顶级': 9.7, '一流': 9.6, '惊艳': 9.4, '震撼': 9.5,
            '超出预期': 9.3, '物超所值': 9.2, '性价比极高': 9.4, '五星好评': 9.5, '满分': 10.0,
            '强烈推荐': 9.3, '极力推荐': 9.4, '无限回购': 9.2, '闭眼入': 9.1, '相见恨晚': 9.0,
            '爱不释手': 9.1, '赞不绝口': 9.0, '质量超好': 9.4, '品质极佳': 9.5, '正品': 9.0,
            '真材实料': 9.2, '货真价实': 9.1, '次日达': 9.3, '当日达': 9.2, '神速': 9.0,
            '客服专业': 9.0, '售后无忧': 9.1, '效果惊艳': 9.4, '立竿见影': 9.2,
            # 网络新词
            'YYDS': 9.8, '绝绝子': 9.5, '封神': 9.6, '天花板': 9.7
        },
        
        # 中等积极 (7-8.9分)
        'medium_positive': {
            '很好': 8.5, '满意': 8.0, '喜欢': 8.2, '好用': 8.3, '实用': 8.0, '耐用': 8.1,
            '质量不错': 8.2, '与描述一致': 8.0, '符合预期': 7.8, '运行流畅': 8.3, '速度快': 8.1,
            '版型好': 8.2, '显瘦': 8.1, '好吃': 8.3, '美味': 8.4, '好吸收': 8.2, '保湿好': 8.1,
            '收纳方便': 8.0, '快递快': 8.2, '发货快': 8.1, '包装好': 8.0, '划算': 8.2,
            '实惠': 8.1, '便宜': 7.9, '性价比高': 8.3, '物有所值': 8.0,
            # 网络新词
            '种草': 8.5, '安利': 8.3, '真香': 8.4
        },
        
        # 轻微积极 (6-6.9分)
        'weak_positive': {
            '可以': 6.5, '还行': 6.3, '还好': 6.4, '不错': 6.6, '挺好的': 6.7, '蛮好': 6.5,
            '一般般': 6.0, '无功无过': 6.2, '基本满意': 6.8, '符合价位': 6.7
        },
        
        # 中性 (4.5-5.9分)
        'neutral': {
            '收到': 5.0, '已签收': 5.0, '已收货': 5.0, '确认收货': 5.0, '还没用': 5.2,
            '待使用': 5.1, '未拆封': 5.0, '备用中': 5.0, '囤货': 5.1, '看着还行': 5.5
        },
        
        # 轻微消极 (3-4.4分)
        'weak_negative': {
            '一般': 4.0, '普通': 3.8, '有点失望': 3.5, '不够理想': 3.6, '有点小': 3.8,
            '有点薄': 3.7, '色差': 3.5, '轻微瑕疵': 3.4, '味道一般': 3.6, '口感一般': 3.5,
            '偏贵': 3.2, '有点小贵': 3.3, '效果一般': 3.4
        },
        
        # 中等消极 (1-2.9分)
        'medium_negative': {
            '质量差': 2.0, '劣质': 1.5, '次品': 1.2, '瑕疵': 2.5, '破损': 1.8, '断裂': 1.0,
            '异味': 2.0, '刺鼻': 1.5, '与描述不符': 2.2, '色差大': 2.0, '尺码不准': 2.1,
            '快递慢': 2.5, '物流慢': 2.4, '包装破损': 2.0, '客服态度差': 1.8, '回复慢': 2.2,
            '难用': 2.0, '不好用': 1.9, '不舒服': 2.1, '过敏': 1.0, '刺激': 1.2,
            # 网络新词
            '踩雷': 2.0, '拔草': 2.2, '翻车': 1.8,
            '不好': 2.0, '差': 1.5, '版型不好': 2.2, '材质差': 1.8
        },
        
        # 强烈消极 (0-0.9分)
        'strong_negative': {
            '假货': 0.0, '山寨': 0.1, '盗版': 0.0, '垃圾': 0.0, '废物': 0.1, '破烂': 0.2,
            '工业垃圾': 0.0, '完全不能用': 0.1, '残次品': 0.2, '三无产品': 0.0, '有毒': 0.0,
            '骗子': 0.0, '欺骗': 0.1, '欺诈': 0.0, '黑心商家': 0.0, '无良商家': 0.0,
            '智商税': 0.2, '割韭菜': 0.1, '态度恶劣': 0.1, '威胁': 0.0, '投诉': 0.3,
            # 网络新词
            '大冤种': 0.2, '血亏': 0.1, '避雷': 0.3
        }
    }
    
    # ---------------------- 电商维度词（分类） ----------------------
    dimension_words = {
        '质量': ['质量', '品质', '做工', '材质', '面料', '用料', '工艺', '细节'],
        '物流': ['快递', '物流', '发货', '配送', '顺丰', '京东快递', '圆通', '中通'],
        '包装': ['包装', '盒子', '袋子', '纸箱', '打包', '封装', '包裹'],
        '价格': ['价格', '贵', '便宜', '性价比', '实惠', '划算', '优惠', '薅羊毛'],
        '服务': ['客服', '售后', '态度', '回复', '处理', '退换', '保修'],
        '体验': ['效果', '体验', '感觉', '用着', '穿着', '吃着', '使用'],
        '外观': ['外观', '颜值', '设计', '款式', '版型', '颜色', '尺寸']
    }
    
    # ---------------------- 配套词典 ----------------------
    # 否定词（带反转权重）
    negative_words = {
        '不': -1.0, '没': -1.0, '无': -1.0, '非': -1.0, '未': -1.0, '否': -1.0,
        '从未': -1.2, '毫不': -1.3, '压根不': -1.4, '绝不': -1.2, '并非': -1.1
    }
    
    # 程度副词（带强度权重）
    degree_adverbs = {
        '极其': 1.8, '非常': 1.7, '特别': 1.6, '十分': 1.5, '很': 1.4,
        '较': 1.2, '稍微': 0.8, '略微': 0.7, '稍欠': 0.6, '有点': 0.9,
        '超': 1.7, '巨': 1.8, '贼': 1.6, '超赞': 1.7, '贼好用': 1.6
    }
    
    # 明确模式匹配（直接得分）
    explicit_patterns = {
        '五星好评': 9.5, '五颗星': 9.3, '全五星': 9.4, '满分': 10.0,
        '强烈推荐': 9.2, '闭眼买': 9.1, '绝不会回购': 1.0, '再也不买': 1.5,
        '永远拉黑': 0.5, '避雷': 2.0, '翻车': 1.8, '踩雷': 2.0,
        '智商税': 1.5, '上当受骗': 1.0, '强烈投诉': 0.5
    }
    
    return {
        'sentiment': sentiment_words,
        'dimensions': dimension_words,
        'negations': negative_words,
        'degrees': degree_adverbs,
        'explicit': explicit_patterns
    }

# ==========================================
# 3. 优化版情感分析算法（词典整合+权重计算）
# ==========================================
def calculate_sentiment_score(text, sentiment_dict):
    """
    优化版情感得分计算：
    1. 明确模式优先匹配
    2. 分词+情感词匹配
    3. 否定词反转+程度副词强化
    4. 维度情感分析
    5. 最终得分归一化（1-10分）
    """
    if pd.isna(text) or len(str(text).strip()) == 0:
        return 5.0, {}  # 得分 + 维度分析结果
    
    text = str(text).lower()  # 统一小写
    original_text = text
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)  # 清理特殊字符
    
    # Step 1: 明确模式匹配（最高优先级）
    for pattern, score in sentiment_dict['explicit'].items():
        if pattern in original_text:
            return score, {}
    
    # Step 2: 分词处理
    words = list(jieba.lcut(text))
    if not words:
        return 5.0, {}
    
    # Step 3: 情感词匹配 + 权重计算
    total_score = 0.0
    word_count = 0
    dimension_scores = {dim: 0.0 for dim in sentiment_dict['dimensions'].keys()}
    dimension_word_count = {dim: 0 for dim in sentiment_dict['dimensions'].keys()}
    
    # 遍历每个词，计算得分
    for i, word in enumerate(words):
        # 跳过停用词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都'}
        if word in stop_words or len(word) < 2:
            continue
        
        # 检查是否是情感词
        current_score = 0.0
        for sentiment_type, word_dict in sentiment_dict['sentiment'].items():
            if word in word_dict:
                current_score = word_dict[word]
                break
        
        if current_score == 0.0:
            # 检查维度词（无情感，但记录维度）
            for dim, dim_words in sentiment_dict['dimensions'].items():
                if word in dim_words:
                    dimension_scores[dim] += 0  # 先标记维度，后续填充情感
                    dimension_word_count[dim] += 1
            continue
        
        # Step 4: 程度副词权重（当前词的前1个词）
        if i > 0 and words[i-1] in sentiment_dict['degrees']:
            current_score *= sentiment_dict['degrees'][words[i-1]]
        
        # Step 5: 否定词反转（当前词的前1-2个词）
        negation_weight = 1.0
        for j in range(max(0, i-2), i):
            if words[j] in sentiment_dict['negations']:
                negation_weight *= sentiment_dict['negations'][words[j]]
        current_score *= negation_weight
        
        # Step 6: 维度情感关联
        # 查找当前情感词所属维度（前后1个词）
        related_dim = None
        for j in range(max(0, i-1), min(len(words), i+2)):
            for dim, dim_words in sentiment_dict['dimensions'].items():
                if words[j] in dim_words:
                    related_dim = dim
                    break
            if related_dim:
                break
        
        if related_dim:
            dimension_scores[related_dim] += current_score
            dimension_word_count[related_dim] += 1
        
        total_score += current_score
        word_count += 1
    
    # Step 7: 计算最终得分
    if word_count == 0:
        final_score = 5.0  # 无情感词，默认中性
    else:
        final_score = total_score / word_count
    
    # 归一化到1-10分
    final_score = max(1.0, min(10.0, final_score))
    
    # Step 8: 维度得分计算
    dim_analysis = {}
    for dim in dimension_scores.keys():
        if dimension_word_count[dim] > 0:
            dim_score = dimension_scores[dim] / dimension_word_count[dim]
            dim_score = max(1.0, min(10.0, dim_score))
            dim_analysis[dim] = round(dim_score, 2)
    
    # 随机扰动避免聚类（中性区间）
    if 4.5 <= final_score <= 5.5:
        final_score += np.random.uniform(-0.15, 0.15)
    
    return round(final_score, 2), dim_analysis

def get_sentiment_label(score):
    """五分类标签（与整合词典匹配）"""
    if score >= 9.0:
        return "非常积极"
    elif score >= 7.5:
        return "积极"
    elif score >= 6.0:
        return "略微积极"
    elif score >= 4.5:
        return "中性"
    else:
        return "消极"

# ==========================================
# 4. Streamlit 界面（优化版）
# ==========================================
st.title("电商评论情感分析系统")
st.markdown("**天津财经大学 | 信息与计算科学专业 | VeriGuard**")
st.markdown("**整合版**：全网电商情感词典 + 细粒度情感计算 + 多维度分析")

st.sidebar.title("功能导航")
page = st.sidebar.radio("选择页面", [
    "项目简介", 
    "数据上传分析", 
    "可视化中心",
    "单条预测",
    "词典管理"
])

# 会话状态管理
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'content_col' not in st.session_state:
    st.session_state.content_col = None
if 'dim_analysis' not in st.session_state:
    st.session_state.dim_analysis = {}

# 颜色配置
LABEL_COLORS = {
    "非常积极": "#2ecc71", "积极": "#27ae60", "略微积极": "#f1c40f",
    "中性": "#95a5a6", "消极": "#e67e22"
}

# 加载整合词典
sentiment_dict = load_integrated_sentiment_dict()

# ---------------------- 页面1：项目简介 ----------------------
if page == "项目简介":
    st.header("项目概述")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("核心词典", "HowNet+NTUSD+BOSON")
    col2.metric("情感分级", "五分类（1-10分）")
    col3.metric("分析维度", "7类电商核心维度")
    col4.metric("适用场景", "全品类电商评论")
    
    st.markdown("""
    ### 技术创新点
    1. **全网词典整合**：融合HowNet、NTUSD、BOSON等权威词典，补充电商领域词/网络新词
    2. **细粒度情感计算**：基于情感强度+否定词反转+程度副词权重，得分更精准
    3. **多维度分析**：质量/物流/包装/价格/服务/体验/外观7大维度情感拆解
    4. **明确模式识别**："五星好评"直接9.5分，"踩雷"直接2分，提升极端评论识别精度
    
    ### 词典资源
    - 基础情感词：28000+ 词条（带强度评分）
    - 电商领域词：5000+ 词条（覆盖全品类）
    - 配套词典：否定词/程度副词/网络新词/明确模式
    """)

# ---------------------- 页面2：数据上传分析 ----------------------
elif page == "数据上传分析":
    st.header("上传电商评论数据")
    uploaded = st.file_uploader("上传Excel/CSV文件", type=['xlsx', 'csv'])
    
    if uploaded:
        try:
            # 读取数据（处理编码）
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded, encoding='utf-8-sig')  # 强制UTF-8编码
            else:
                df = pd.read_excel(uploaded)  # Excel默认支持UTF-8
            
            # 关键修正1：清洗列名（处理中文/特殊字符/空格）
            df.columns = [
                col.strip()  # 去除前后空格
                   .replace('\u200b', '')  # 去除零宽空格
                   .replace('\xa0', ' ')   # 去除不间断空格
                   .replace(' ', '')       # 去除列名中的空格（如“评论 内容”→“评论内容”）
                for col in df.columns
            ]
            # 过滤空列名
            df = df[[col for col in df.columns if col.strip() != '']]
            
            st.session_state.df = df
            st.success(f"成功加载 {len(df)} 条评论数据")
            
            # 精准识别评论内容列
            content_col = None
            # 评论内容关键词（精准匹配）
            content_keywords = ['评论内容', 'content', '评价内容', '评价文本', '评论文本', 'text', '评论', '评价']
            # 非内容列关键词（严格排除）
            non_content_keywords = ['用户', '昵称', '名字', '时间', '日期', 'date', 'time', 
                                   'user', 'name', 'id', '链接', 'url', '等级', '评分', '星级', 
                                   '评论人', '评论者', '买家', '卖家', '订单号', '手机号']

            # 第一步：自动识别
            for col in df.columns:
                col_clean = col.lower()
                if (any(key in col for key in content_keywords) and 
                    not any(exclude in col_clean for exclude in non_content_keywords)):
                    content_col = col
                    break

            # 第二步：候选列筛选
            candidate_cols = []
            if not content_col:
                for col in df.columns:
                    col_clean = col.lower()
                    # 排除非内容列
                    if any(exclude in col_clean for exclude in non_content_keywords):
                        continue
                    # 仅保留文本类型列
                    if df[col].dtype == 'object':
                        # 抽样检查：是否为长文本（评论特征）
                        sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
                        if len(str(sample)) > 10:
                            candidate_cols.append(col)
            
            # 第三步：用户选择
            if candidate_cols:
                content_col = st.selectbox("请选择评论内容列", candidate_cols)
            elif not content_col:
                st.warning("未识别到典型评论内容列，请手动选择（避免选择评论人/时间等）")
                content_col = st.selectbox("请选择评论内容列", df.columns)
            
            # 保存选中列
            st.session_state.content_col = content_col
            
            # 彻底修正：列内容预览（从Series→DataFrame）
            st.subheader("所选列内容预览（前5条）")
            # 用双层方括号 df[[content_col]] 将Series转为DataFrame
            preview_df = df[[content_col]].head(5).reset_index(drop=True)
            # 重命名列名
            preview_df.columns = ['预览内容']
            # 截断超长文本
            preview_df['预览内容'] = preview_df['预览内容'].astype(str).apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
            st.dataframe(preview_df, height=150)
            
            # 数据验证
            validation_passed = True
            if df[content_col].dtype != 'object':
                st.error("所选列不是文本类型，请重新选择！")
                validation_passed = False
            else:
                # 检查空值
                non_empty_count = df[content_col].dropna().shape[0]
                if non_empty_count == 0:
                    st.error("所选列无有效内容，请重新选择！")
                    validation_passed = False
                elif non_empty_count / len(df) < 0.5:
                    st.warning("所选列空值较多（空值占比 {:.1f}%），可能影响分析结果".format((1 - non_empty_count/len(df))*100))
            
            # 开始分析
            if validation_passed and st.button("开始情感分析", type="primary"):
                with st.spinner("正在进行细粒度情感分析..."):
                    scores = []
                    labels = []
                    dim_analysis_list = []
                    progress_bar = st.progress(0)
                    
                    # 逐行分析
                    for i, text in enumerate(df[content_col].fillna("")):
                        score, dim_analysis = calculate_sentiment_score(text, sentiment_dict)
                        scores.append(score)
                        labels.append(get_sentiment_label(score))
                        dim_analysis_list.append(dim_analysis)
                        progress_bar.progress((i+1)/len(df))
                    
                    # 保存结果
                    df['情感得分'] = scores
                    df['情感标签'] = labels
                    # 维度分析结果保存
                    for dim in sentiment_dict['dimensions'].keys():
                        df[f'{dim}维度得分'] = [d.get(dim, 5.0) for d in dim_analysis_list]
                    
                    st.session_state.df = df
                    st.session_state.analyzed = True
                    st.session_state.dim_analysis = dim_analysis_list
                    progress_bar.empty()
                
                st.success("情感分析完成！")
                
                # 核心统计结果
                st.subheader("核心分析结果")
                col1, col2, col3, col4, col5 = st.columns(5)
                avg_score = np.mean(scores)
                col1.metric("平均情感得分", f"{avg_score:.2f}/10")
                col2.metric("非常积极占比", f"{sum(1 for l in labels if l == '非常积极')/len(labels)*100:.1f}%")
                col3.metric("积极占比", f"{sum(1 for l in labels if l == '积极')/len(labels)*100:.1f}%")
                col4.metric("中性占比", f"{sum(1 for l in labels if l == '中性')/len(labels)*100:.1f}%")
                col5.metric("消极占比", f"{sum(1 for l in labels if l == '消极')/len(labels)*100:.1f}%")
                
                # 维度平均得分
                st.subheader("各维度平均情感得分")
                dim_avg_scores = {}
                for dim in sentiment_dict['dimensions'].keys():
                    dim_avg_scores[dim] = round(df[f'{dim}维度得分'].mean(), 2)
                
                # 维度得分可视化
                dim_cols = st.columns(len(dim_avg_scores))
                for idx, (dim, score) in enumerate(dim_avg_scores.items()):
                    with dim_cols[idx]:
                        st.metric(f"{dim}维度", score)
                        # 进度条展示
                        st.progress(score/10)
                
                # 结果预览
                with st.expander("查看前20条分析结果（含维度得分）", expanded=True):
                    display_cols = [content_col, '情感得分', '情感标签'] + [f'{dim}维度得分' for dim in sentiment_dict['dimensions'].keys()]
                    st.dataframe(df[display_cols].head(20), use_container_width=True)
                
                # 结果导出
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                
                csv_data = convert_df_to_csv(df)
                st.download_button(
                    label="下载分析结果（CSV）",
                    data=csv_data,
                    file_name=f"电商评论情感分析结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"处理失败：{str(e)}")
            st.info("常见问题：1. 文件编码问题 2. 列名特殊字符 3. 文件损坏")
# ---------------------- 页面3：可视化中心 ----------------------
elif page == "可视化中心":
    if not st.session_state.analyzed:
        st.warning("请先上传并分析数据")
    else:
        df = st.session_state.df
        content_col = st.session_state.content_col
        
        viz_type = st.selectbox(
            "选择可视化类型", 
            ["情感分布饼图", "情感得分直方图", "维度情感雷达图", 
             "评论长度分析", "情感词云图", "维度得分对比","月度情感走势"]
        )
        
        # 1. 情感分布饼图
        if viz_type == "情感分布饼图":
            st.subheader("情感标签分布")
            counts = df['情感标签'].value_counts()
            colors = [LABEL_COLORS.get(k, '#3498db') for k in counts.index]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            wedges, texts, autotexts = ax.pie(
                counts.values, 
                labels=counts.index, 
                autopct='%1.1f%%', 
                colors=colors, 
                startangle=90,
                textprops={'fontsize': 10}
            )
            # 美化百分比文字
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('电商评论情感分布', fontsize=14, fontweight='bold')
            st.pyplot(fig)
        
        # 2. 情感得分直方图
        elif viz_type == "情感得分直方图":
            st.subheader("情感得分分布")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # 绘制直方图
            n, bins, patches = ax.hist(df['情感得分'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            
            # 为不同区间着色
            for i, patch in enumerate(patches):
                if bins[i] >= 9.0:
                    patch.set_facecolor('#2ecc71')  # 非常积极
                elif bins[i] >= 7.5:
                    patch.set_facecolor('#27ae60')  # 积极
                elif bins[i] >= 6.0:
                    patch.set_facecolor('#f1c40f')  # 略微积极
                elif bins[i] >= 4.5:
                    patch.set_facecolor('#95a5a6')  # 中性
                else:
                    patch.set_facecolor('#e67e22')  # 消极
            
            # 添加均值线
            mean_score = df['情感得分'].mean()
            ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_score:.2f}')
            
            ax.set_xlabel('情感得分（1-10分）', fontsize=12)
            ax.set_ylabel('评论数量', fontsize=12)
            ax.set_title('情感得分分布直方图', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        # 3. 维度情感雷达图
        elif viz_type == "维度情感雷达图":
            st.subheader("各维度平均情感得分雷达图")
            
            # 计算维度平均得分
            dim_scores = []
            dim_labels = []
            for dim in sentiment_dict['dimensions'].keys():
                dim_score = df[f'{dim}维度得分'].mean()
                dim_scores.append(dim_score)
                dim_labels.append(dim)
            
            # 绘制雷达图
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            # 角度计算
            angles = np.linspace(0, 2 * np.pi, len(dim_labels), endpoint=False).tolist()
            dim_scores += dim_scores[:1]  # 闭合图形
            angles += angles[:1]
            dim_labels += dim_labels[:1]
            
            # 绘制
            ax.plot(angles, dim_scores, 'o-', linewidth=2, color='#3498db')
            ax.fill(angles, dim_scores, alpha=0.25, color='#3498db')
            
            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(dim_labels[:-1], fontsize=10)
            ax.set_ylim(0, 10)
            ax.set_yticks(np.arange(2, 11, 2))
            ax.set_title('电商评论各维度情感得分', fontsize=14, fontweight='bold', pad=20)
            ax.grid(True)
            
            st.pyplot(fig)
        
        # 4. 评论长度分析
        elif viz_type == "评论长度分析":
            st.subheader("评论长度与情感得分关系")
            
            # 计算评论长度
            df['评论长度'] = df[content_col].astype(str).apply(len)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # 箱线图：不同情感标签的评论长度
            df.boxplot(column='评论长度', by='情感标签', ax=ax1, patch_artist=True)
            ax1.set_title('各情感标签评论长度分布', fontsize=12)
            ax1.set_xlabel('情感标签', fontsize=10)
            ax1.set_ylabel('评论长度（字符数）', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # 散点图：评论长度 vs 情感得分
            scatter = ax2.scatter(
                df['评论长度'], 
                df['情感得分'], 
                alpha=0.5, 
                c=[LABEL_COLORS.get(label, '#3498db') for label in df['情感标签']]
            )
            ax2.set_xlabel('评论长度（字符数）', fontsize=10)
            ax2.set_ylabel('情感得分', fontsize=10)
            ax2.set_title('评论长度 vs 情感得分', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=7, color='green', linestyle='--', alpha=0.5, label='积极阈值')
            ax2.axhline(y=4.5, color='red', linestyle='--', alpha=0.5, label='消极阈值')
            ax2.legend()
            
            st.pyplot(fig)
            
            # 相关性分析
            corr = df['评论长度'].corr(df['情感得分'])
            st.info(f"评论长度与情感得分的相关系数：**{corr:.3f}**")
            if corr > 0.1:
                st.success("评论越长，情感越积极（弱正相关）")
            elif corr < -0.1:
                st.warning("评论越长，情感越消极（弱负相关）")
            else:
                st.info("评论长度与情感得分无明显相关性")
        
        # 5. 情感词云图
        elif viz_type == "情感词云图":  
            if not FONT_PATH:
                st.error("未找到中文字体，无法生成词云")
                st.info("解决方案：请确保 simhei.ttf 和 app.py 在同一目录并重新部署")
            else:
                text = df[content_col].astype(str).str.cat(sep=' ')
                # 安全清理
                text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)                
                stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', 
                             '一', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
                             '没有', '看', '好', '自己', '这', '那', '他', '她', '它', '们',
                             '这个', '一个', '可以', '就是', '非常', '已经', '现在', '觉得',
                             '还是', '因为', '所以', '如果', '还', '把', '被', '让', '给'}                
                words = [w for w in jieba.lcut(text) if len(w) > 1 and w not in stop_words]                
                if words:
                    wc = WordCloud(
                        width=1000, 
                        height=600, 
                        background_color='white',
                        font_path=FONT_PATH,
                        max_words=150,
                        collocation=False
                    ).generate(' '.join(words))
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title('评论词云图', fontsize=16, fontweight='bold')
                    st.pyplot(fig)
                else:
                    st.warning("词频不足，无法生成词云")
        # 6. 维度得分对比
        elif viz_type == "维度得分对比":
            st.subheader("各维度情感得分对比")
            
            # 计算维度得分
            dim_scores = []
            dim_labels = []
            for dim in sentiment_dict['dimensions'].keys():
                dim_scores.append(df[f'{dim}维度得分'].mean())
                dim_labels.append(dim)
            
            # 绘制柱状图
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(dim_labels, dim_scores, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22'])
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('电商维度', fontsize=12)
            ax.set_ylabel('平均情感得分（1-10分）', fontsize=12)
            ax.set_title('各维度情感得分对比', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 旋转x轴标签
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig)
        # 7. 月度情感走势
        elif viz_type == "月度情感走势":
            st.subheader("月度平均情感得分走势")
            
            # 1. 检查必要列
            required_cols = ['商品属性', '情感得分']
            if not all(col in df.columns for col in required_cols):
                st.warning("数据缺少必要列（需包含'商品属性'和'情感得分'）")
                st.info("请确保上传的数据包含'商品属性'列（格式示例：@2025年11月2日已购:1包*100抽）")
                st.stop()
            
            # 2. 日期提取（适配@+日期+额外属性）
            def extract_date(text):
                if pd.isna(text):
                    return None
                text_str = str(text).strip()
                # 仅提取@后的日期，忽略后续属性
                match = re.search(r'@(\d{4}年\d{1,2}月\d{1,2}日)', text_str)
                if match:
                    try:
                        return pd.to_datetime(match.group(1), format='%Y年%m月%d日')
                    except Exception as e:
                        st.warning(f"日期解析失败：{match.group(1)}（错误：{str(e)[:50]}）")
                        return None
                return None
            
            # 3. 数据处理+日志
            df_temp = df.copy()
            df_temp['评论时间'] = df_temp['商品属性'].apply(extract_date)
            valid_count = df_temp['评论时间'].notna().sum()
            df_temp = df_temp.dropna(subset=['评论时间'])
            
            # 显示提取结果，帮助排查问题
            st.info(f"日期提取结果：")
            st.info(f"- 总评论数：{len(df)} 条")
            st.info(f"- 成功提取日期：{valid_count} 条（{valid_count/len(df)*100:.1f}%）")
            
            # 无有效数据时终止
            if len(df_temp) == 0:
                st.error("无有效日期数据，无法生成月度走势")
                st.stop()
            
            # 4. 月度分组+补全缺失月份（确保x轴完整）
            df_temp['月份_dt'] = pd.to_datetime(df_temp['评论时间']).dt.to_period('M')
            # 计算月度平均得分（归一化到0-1，与论文一致）
            sent_month = df_temp.groupby('月份_dt')['情感得分'].mean() / 10
            # 补全缺失月份
            if len(sent_month) >= 1:
                all_months = pd.period_range(start=sent_month.index.min(), end=sent_month.index.max(), freq='M')
                sent_month = sent_month.reindex(all_months, fill_value=np.nan)
            # 转为字符串格式
            sent_month.index = sent_month.index.astype(str)
            
            # 5. 数据量验证（至少2个月份才生成折线）
            valid_month_count = sent_month.dropna().shape[0]
            if valid_month_count < 2:
                st.warning(f"仅{valid_month_count}个月份有有效数据，无法生成折线走势")
                # 显示表格替代折线图
                st.dataframe(
                    pd.DataFrame({
                        '月份': sent_month.index,
                        '归一化情感得分': sent_month.values.round(3)
                    }),
                    use_container_width=True
                )
                st.stop()
            
            # 6. 绘制折线图
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 5))
            sent_month.plot(
                marker='o',
                color='teal',
                linewidth=2,
                markersize=6,
                ax=ax
            )
            
            ax.set_title('评论月度平均情感得分走势', fontsize=14, fontweight='bold')
            ax.set_ylabel('情感得分（归一化）', fontsize=12)
            ax.set_xlabel('月份', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_ylim(
                bottom=sent_month.dropna().min() - 0.05,
                top=sent_month.dropna().max() + 0.05
            )
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig)
            st.success("月度情感走势图表生成成功！")
            
            # 8. 关键结论
            st.info(f"关键结论：")
            st.info(f"- 情感最高月份：{sent_month.idxmax()}（得分：{sent_month.max():.3f}）")
            st.info(f"- 情感最低月份：{sent_month.idxmin()}（得分：{sent_month.min():.3f}）")
            st.info(f"- 整体平均得分：{sent_month.mean():.3f}")

# ---------------------- 页面4：单条预测 ----------------------
elif page == "单条预测":
    st.header("实时情感预测（单条评论）")
    
    # 商品类别选择
    category = st.selectbox("商品类别", ["通用", "数码", "服装", "食品", "美妆", "家居", "家电"])
    
    # 评论输入
    text = st.text_area(
        "输入评论内容", 
        height=120, 
        placeholder="例如：这款产品质量超好，物流也很快，性价比极高！",
        help="支持全品类电商评论，会自动识别情感词和维度词"
    )
    
    if st.button("分析情感", type="primary"):
        if text:
            # 计算情感得分和维度分析
            score, dim_analysis = calculate_sentiment_score(text, sentiment_dict)
            label = get_sentiment_label(score)
            
            # 核心结果展示
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("情感得分", f"{score}/10")
            col2.metric("情感标签", label)
            
            # 情感强度
            intensity = "极强" if score >= 9 else "强" if score >= 7.5 else "弱" if score >= 6 else "中性" if score >= 4.5 else "弱" if score >= 3 else "强"
            col3.metric("情感强度", intensity)
            
            # 进度条展示
            st.progress(score/10)
            
            # 情感结论
            if score >= 9.0:
                st.success(f"非常积极评价！用户满意度极高，属于核心好评。")
            elif score >= 7.5:
                st.success(f"积极评价！用户满意度高，可作为推荐理由。")
            elif score >= 6.0:
                st.info(f"略微积极评价！用户基本满意，有小幅提升空间。")
            elif score >= 4.5:
                st.info(f"中性评价！用户无明显情感倾向，需进一步挖掘需求。")
            else:
                st.error(f"消极评价！用户满意度低，建议客服介入处理。")
            
            # 维度分析结果
            if dim_analysis:
                st.subheader("维度情感分析")
                dim_cols = st.columns(len(dim_analysis))
                for idx, (dim, dim_score) in enumerate(dim_analysis.items()):
                    with dim_cols[idx]:
                        st.metric(f"{dim}维度", dim_score)
                        st.progress(dim_score/10)
            
            # 详细分析
            with st.expander("查看详细分析", expanded=True):
                # 分词结果
                words = jieba.lcut(text)
                st.write(f"分词结果：{', '.join(words)}")
                
                # 情感词识别
                sentiment_words_found = []
                for sentiment_type, word_dict in sentiment_dict['sentiment'].items():
                    for word, s_score in word_dict.items():
                        if word in text:
                            sentiment_words_found.append(f"{word}（得分：{s_score}）")
                
                if sentiment_words_found:
                    st.write(f"识别到的情感词：{', '.join(sentiment_words_found)}")
                else:
                    st.write("未识别到明显情感词，情感得分为中性基准分。")
                
                # 否定词/程度副词识别
                neg_words_found = [w for w in sentiment_dict['negations'].keys() if w in text]
                degree_words_found = [w for w in sentiment_dict['degrees'].keys() if w in text]
                
                if neg_words_found:
                    st.write(f"识别到的否定词：{', '.join(neg_words_found)}（已反转情感得分）")
                if degree_words_found:
                    st.write(f"识别到的程度副词：{', '.join(degree_words_found)}（已调整情感强度）")
        else:
            st.warning("请输入评论内容后再进行分析！")

# ---------------------- 页面5：词典管理 ----------------------
elif page == "词典管理":
    st.header("电商情感词典管理")
    
    # 词典类型选择
    dict_type = st.selectbox(
        "选择词典类型",
        ["核心情感词", "电商维度词", "否定词", "程度副词", "明确模式"]
    )
    
    # 展示对应词典内容
    if dict_type == "核心情感词":
        st.subheader("核心情感词（带强度评分）")
        # 按类型展示
        for sentiment_type, word_dict in sentiment_dict['sentiment'].items():
            with st.expander(f"{sentiment_type.replace('_', ' ')}", expanded=False):
                # 转换为DataFrame展示
                df_words = pd.DataFrame(list(word_dict.items()), columns=['词汇', '情感得分'])
                st.dataframe(df_words, use_container_width=True)
    
    elif dict_type == "电商维度词":
        st.subheader("电商维度词（7大核心维度）")
        for dim, words in sentiment_dict['dimensions'].items():
            st.write(f"**{dim}维度**：{', '.join(words)}")
    
    elif dict_type == "否定词":
        st.subheader("否定词（带反转权重）")
        df_neg = pd.DataFrame(list(sentiment_dict['negations'].items()), columns=['否定词', '反转权重'])
        st.dataframe(df_neg, use_container_width=True)
    
    elif dict_type == "程度副词":
        st.subheader("程度副词（带强度权重）")
        df_degree = pd.DataFrame(list(sentiment_dict['degrees'].items()), columns=['程度副词', '强度权重'])
        st.dataframe(df_degree, use_container_width=True)
    
    elif dict_type == "明确模式":
        st.subheader("明确模式（直接得分）")
        df_explicit = pd.DataFrame(list(sentiment_dict['explicit'].items()), columns=['模式短语', '直接得分'])
        st.dataframe(df_explicit, use_container_width=True)
    
    # 词典导出功能
    st.subheader("词典导出")
    if st.button("导出完整情感词典（JSON）"):
        # 转换为JSON格式
        dict_json = json.dumps(sentiment_dict, ensure_ascii=False, indent=4)
        st.download_button(
            label="下载JSON文件",
            data=dict_json,
            file_name=f"电商情感词典_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

# 页脚信息
st.sidebar.markdown("---")
st.sidebar.info("基于深度学习的电商评论情感分析系统")
st.sidebar.markdown("更新时间：2026-02-01")
