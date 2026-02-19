from langchain_core.documents import Document

chunk = Document(
    page_content=('제21조(약관교부 및 설명의무 등)\n'
 '\uf000 회사는 계약자가 청약할 때에 계약자에게 약관의 중요한 내용을 설명하여야 하며, 청약 후에 다음 각 호의 방법 중 계약자가 '
 '원하는 방법을 확인하여 지체 없이 약관 및 계약 자 보관용 청약서를 제공하여 드립니다. 만약, 회사가 전자 우편 및 전자적 의사표시로 '
 '제공한 경우 계약자 또는 그 대 리인이 약관 및 계약자 보관용 청약서 등을 수신하였을 때 에는 해당 문서를 드린 것으로 봅니다.\n'
 '① 서면교부 ② 우편 또는 전자우편 ③ 휴대전화 문자메시지 또는 이에 준하는 전자적 의사표시'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 69},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000088',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
