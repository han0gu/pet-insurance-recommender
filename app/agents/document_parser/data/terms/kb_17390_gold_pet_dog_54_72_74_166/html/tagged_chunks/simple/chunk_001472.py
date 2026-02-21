from langchain_core.documents import Document

chunk = Document(
    page_content=('신경계․정신행동의 13개 부위를 말하며, 이를 각각 동일한 신체부위라 한다.<br>다만, 좌․우의 눈, 귀, 팔, 다리, 손가락, '
 "발가락은 각각 다른 신체부위로 본다.</p><br><h1 id='162' style='font-size:14px'>3"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'eye', 'head']},
 'indexing': {'chunk_id': 'chunk_001472',
              'chunk_char_len': 138,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
