from langchain_core.documents import Document

chunk = Document(
    page_content=('여 체납된 세금에 대하여 가산금 징수, 독촉장 발부 및 재산 압류 등의 집행을 하는 것을 말합니 다. 국세 및 지방세 체납시 국세청 및 '
 '지방자치단체에 의해 채무자의 해약환급금이 압류될 수 있으며, 체납처분 절차에 따라 회사는 채권자에게 해약환급금을 지급하게 됩니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 105},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000625',
              'chunk_char_len': 148,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
