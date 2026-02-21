from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항에서 보험증권을 받은 날에 대한 다툼이 발생한 경\n'
 '우 회사가 이를 증명하여야 합니다.제21조(약관교부 및 설명의무 등)\uf000 회사는 계약자가 청약할 때에 계약자에게 약관의 중요한\n'
 '내용을 설명하여야 하며, 청약 후에 다음 각 호의 방법 중\n'
 '계약자가 원하는 방법을 확인하여 지체 없이 약관 및 계약\n'
 '자 보관용 청약서를 제공하여 드립니다. 만약, 회사가 전자\n'
 '우편 및 전자적 의사표시로 제공한 경우 계약자 또는 그 대\n'
 '리인이 약관 및 계약자 보관용 청약서 등을 수신하였을 때\n'
 '에는 해당 문서를 드린 것으로 봅니다.- ① 서면교부'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000072',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
