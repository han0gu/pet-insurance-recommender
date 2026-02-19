from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항에도 불구하고 청약한 날부터 30일(만 65세 이상의 계약자가 전화를 이용하여 체결한 계약은 45일로 합니다)이 초과된 '
 '계약은 청약을 철회할 수 없습니다. \uf000 청약철회는 계약자가 전화로 신청하거나, 철회의사를 표 시하기 위한 서면, 전자우편, '
 '휴대전화 문자메시지 또는 이 에 준하는 전자적 의사표시(이하‘서면 등’이라 합니다)를 발송한 때 효력이 발생합니다. 계약자는 서면 등을 '
 '발송한 때에 그 발송 사실을 회사에 지체없이 알려야 합니다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 68},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000084',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
