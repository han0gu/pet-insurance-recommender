from langchain_core.documents import Document

chunk = Document(
    page_content=('# 제17조(약관 교부 및 설명의무 등)① 회사는 계약자가 청약할 때에 계약자에게 약관의 중요한 내용을 설명하여야 하며, 청약 후에 '
 '다음\n'
 '각 호의 방법 중 계약자가 원하는 방법을 확인하여 지체 없이 약관 및 계약자 보관용 청약서를 제- 11 -당신에게 좋은보험 삼성화재공하여 '
 '드립니다. 만약, 회사가 전자우편 및 전자적 의사표시로 제공한 경우 계약자 또는 그 대리\n'
 '인이 약관 및 계약자 보관용 청약서 등을 수신하였을 때에는 해당 문서를 드린 것으로 봅니다.- 1. 서면교부\n'
 '- 2. 우편 또는 전자우편'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000045',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
