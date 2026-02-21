from langchain_core.documents import Document

chunk = Document(
    page_content=('- 권리를 행사할 수 있습니다.\n'
 '【제척기간】 어떤 종류의 권리에 대하여 법률상으로 정하여진 존속기간을 말하며, 이 기간이 지나면 해당\n'
 '권리는 소멸됩니다.# 제30조(보험료의 환급)① 이 계약이 무효, 효력상실 또는 해지된 때에는 다음과 같이 보험료를 돌려드립니다.1. '
 '계약자 또는 피보험자의 책임 없는 사유에 의하는 경우 : 무효의 경우에는 회사에 납입한 보험료'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000078',
              'chunk_char_len': 202,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
