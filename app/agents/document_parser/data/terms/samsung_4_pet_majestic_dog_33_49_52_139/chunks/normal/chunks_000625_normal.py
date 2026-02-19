from langchain_core.documents import Document

chunk = Document(
    page_content=('[국세 및 지방세 체납처분 절차]\n'
 '국세 및 지방세 체납처분 절차란 국세 또는 지방세를 체납할 경우 국세 기본법 및 지방세법에 의하 여 체납된 세금에 대하여 가산금 징수, '
 '독촉장 발부 및 재산 압류 등의 집행을 하는 것을 말합니 다 . 국세 및 지방세 체납시 국세청 및 지방자치단체에 의해 채무자의 '
 '해약환급금이 압류될 수 있으며, 체납처분 절차에 따라 회사는 채권자에게 해약환급금을 지급하게 됩니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 107},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000625',
              'chunk_char_len': 222,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
