from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[강제집행과 담보권실행]\n'
 '강제집행이란 사법상 또는 행정법상의 의무를 이행하지 않는 사람에 대하여 국가가 강제 권력으로 그 의무를 이행하는 것을 말합니다. '
 '담보권실행이란 담보권을 설정한 채권자가 채무를 이행하지 않는 채무자에 대하여 해당 담보권을 실행하는 것을 말합니다. 법원은 채권자의 '
 '신청에 따른 강제집행 및 담보권실행으로 채무자의 해약환급금을 압류할 수 있으 며 , 법원의 추심명령 또는 전부명령에 따라 회사는 '
 '채권자에게 해약환급금을 지급하게 됩니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 43},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000141',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
