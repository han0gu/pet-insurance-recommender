from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 청약철회는 계약자가 전화로 신청하거나, 철회의사를 표시하기 위한 서면, 전자우편, 휴대전화 문자메시지 또는 이에 준하는 전자적 '
 '의사표시(이하 ‘서면 등’이라 합니 다)를 발송한 때 효력이 발생합니다. 계약자는 서면 등을 발송한 때에 그 발송 사실을 회사에 지체없이 '
 '알려야 합니다. ④ 계약자가 청약을 철회한 때에는 회사는 청약의 철회를 접수한 날부터 3영업일 이내에 납입한 보험료를 계약자에게 돌려 '
 '드리며, 보험료 반환이 늦어진 기간에 대하여는 이 계약의 보험계약대출이율을 연단위 복리로 계산한 금액을 더하여 지급합니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 41},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000095',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
