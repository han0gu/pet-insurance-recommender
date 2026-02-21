from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이하 같다]하거나 동물의 질병을 예방하는 업(業)을 말한다. 4. "동물병원"이란 동물진료업을 하는 장소로서 제17조에 따른 신고를 '
 '한 진료 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000462',
              'chunk_char_len': 82,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
