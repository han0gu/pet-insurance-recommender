from langchain_core.documents import Document

chunk = Document(
    page_content=('【통신판매계약】\n'
 '전화·우편·인터넷 등 통신수단을 이용하여 체결하는 계약을 말합니다.\n'
 '\uf000 회사가 제1항에 따라 제공될 약관 및 계약자 보관용 청 약서를 청약할 때 계약자에게 전달하지 않거나 약관의 중요 한 내용을 '
 '설명하지 않은 때 또는 계약을 체결할 때 계약자 가 청약서에 자필서명을 하지 않은 때에는 계약자는 계약이 성립한 날부터 3개월 이내에 '
 '계약을 취소할 수 있습니다.\n'
 '【자필서명】'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 70},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000092',
              'chunk_char_len': 215,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
