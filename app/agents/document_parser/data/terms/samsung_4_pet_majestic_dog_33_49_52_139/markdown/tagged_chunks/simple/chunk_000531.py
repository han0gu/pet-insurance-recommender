from langchain_core.documents import Document

chunk = Document(
    page_content=('- 한 채권이 소멸되는 집행법원의 결정\n'
 '# [국세 및 지방세 체납처분 절차]국세 및 지방세 체납처분 절차란 국세 또는 지방세를 체납할 경우 국세 기본법 및 지방세법에 의하\n'
 '여 체납된 세금에 대하여 가산금 징수, 독촉장 발부 및 재산 압류 등의 집행을 하는 것을 말합니\n'
 '다 .\n'
 '국세 및 지방세 체납시 국세청 및 지방자치단체에 의해 채무자의 해약환급금이 압류될 수 있으며,\n'
 '체납처분 절차에 따라 회사는 채권자에게 해약환급금을 지급하게 됩니다.- ② 회사는 제1항에 따른 계약자 명의변경 신청 및 특별약관의 '
 '특별부활(효력회복) 청약을'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000531',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
