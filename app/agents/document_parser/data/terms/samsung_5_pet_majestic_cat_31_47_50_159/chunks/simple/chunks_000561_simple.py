from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이하 같다]하거나 동물의 질병을 예방하는 업(業)을 말한다. 3의2. "동물보건사"란 동물병원 내에서 수의사의 지도 아래 동물의 간호 '
 '또는 진료 보조 업무에 종 사하는 사람으로서 농림축산식품부장관의 자격인정을 받은 사람을 말한다. 4. "동물병원"이란 동물진료업을 하는 '
 '장소로서 제17조에 따른 신고를 한 진료기관을 말한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 99},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000561',
              'chunk_char_len': 183,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
