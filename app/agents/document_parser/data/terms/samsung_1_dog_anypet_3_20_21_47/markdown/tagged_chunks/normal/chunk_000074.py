from langchain_core.documents import Document

chunk = Document(
    page_content=('- 증명한 경우에는 보상하여 드립니다.\n'
 '- ⑥ 회사는 다른 보험가입내역에 대한 계약 전·후 알릴 의무 위반을 이유로 계약을 해지하거나 보험금\n'
 '- 지급을 거절하지 않습니다.\n'
 '- 16 -당신에게 좋은보험 삼성화재# 제27조(중대사유로 인한 해지)① 회사는 아래와 같은 사실이 있을 경우에는 그 사실을 안 날부터 '
 '1개월 이내에 계약을 해지할 수 있\n'
 '습니다.- 1. 계약자 또는 피보험자가 보험금을 지급받을 목적으로 고의로 보험금 지급사유를 발생시킨 경우'),
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
 'indexing': {'chunk_id': 'chunk_000074',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
