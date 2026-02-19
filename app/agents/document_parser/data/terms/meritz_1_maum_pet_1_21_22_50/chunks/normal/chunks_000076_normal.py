from langchain_core.documents import Document

chunk = Document(
    page_content=('② 제1항에도 불구하고 청약한 날부터 30일이 초과된 계약은 청약을 철회할 수 없습니다. ③ 청약철회는 계약자가 전화로 신청하거나, '
 '철회의사를 표시하기 위한 서면, 전자우편, 휴 대전화 문자메시지 또는 이에 준하는 전자적 의사표시(이하 ‘서면 등’이라 합니다)를 발 '
 '송한 때 효력이 발생합니다. 계약자는 서면 등을 발송한 때에 그 발송 사실을 회사에 지체없이 알려야 합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 12},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000076',
              'chunk_char_len': 208,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
