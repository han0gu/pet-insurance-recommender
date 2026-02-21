from langchain_core.documents import Document

chunk = Document(
    page_content=('- 니다.\n'
 '# 제21조(약관 교부 및 설명의무 등)① 회사는 계약자가 청약할 때에 계약자에게 약관의 중요한 내용을 설명하여야 하며, 청약\n'
 '후에 다음 각 호의 방법 중 계약자가 원하는 방법을 확인하여 지체 없이 약관 및 계약\n'
 '자 보관용 청약서를 제공하여 드립니다. 만약, 회사가 전자우편 및 전자적 의사표시로\n'
 '제공한 경우 계약자 또는 그 대리인이 약관 및 계약자 보관용 청약서 등을 수신하였을\n'
 '때에는 해당 문서를 드린 것으로 봅니다.- 1. 서면교부\n'
 '- 2. 우편 또는 전자우편'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000073',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
