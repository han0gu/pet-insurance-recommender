from langchain_core.documents import Document

chunk = Document(
    page_content=('한 때에는 그러하지 아니하다.\n'
 '< 「상법」 제651조의2(서면에 의한 질문의 효력)>\n'
 '보험자가 서면으로 질문한 사항은 중요한 사항으로 추정한다.# 제13조(계약 후 알릴 의무)① 계약을 맺은 후 보험목적에 아래와 같은 '
 '사실이 생긴 경우에는 계약자나 피보험자는 지체없이 서\n'
 '면으로 회사에 알리고 보험증권에 확인을 받아야 합니다.- 1. 청약서의 기재사항을 변경하고자 할 때 또는 변경이 생겼음을 알았을 때\n'
 '- 2. 이 계약에서 보장하는 위험과 동일한 위험을 보장하는 계약을 다른 보험자와 체결하고자 할 때'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000035',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
