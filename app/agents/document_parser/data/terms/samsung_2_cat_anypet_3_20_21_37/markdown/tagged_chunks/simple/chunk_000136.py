from langchain_core.documents import Document

chunk = Document(
    page_content=('되는 보험# 【용어해설】# <소득세법 제59조의4(특별세액공제)>근로소득이 있는 거주자(일용근로자는 제외한다. 이하 이 조에서 같다)가 '
 '해당 과세기간에 만기에 환급\n'
 '되는 금액이 납입보험료를 초과하지 아니하는 보험의 보험계약에 따라 지급하는 다음 각 호의 보험료\n'
 '를 지급한 경우 그 금액의 100분의 12(제1호의 경우에는 100분의 15)에 해당하는 금액을 해당 과세기\n'
 '간의 종합소득산출세액에서 공제한다. 다만, 다음 각 호의 보험료별로 그 합계액이 각각 연 100만원을'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000136',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
