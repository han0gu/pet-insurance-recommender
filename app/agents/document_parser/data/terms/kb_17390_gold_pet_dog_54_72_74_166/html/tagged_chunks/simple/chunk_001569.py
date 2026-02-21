from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>남긴 때</h1><p id='70' data-category='list'></p><br><p "
 "id='71' data-category='paragraph' style='font-size:14px'>나.</p><br><p "
 "id='72' data-category='paragraph' style='font-size:14px'>장해판정기준<br>1) "
 '‘체간골’이라 함은 어깨뼈(견갑골), 골반뼈(장골, 제2천추 이하의 천<br>골, 미골, 좌골 포함), 빗장뼈(쇄골), 가슴뼈(흉골),'),
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
 'indexing': {'chunk_id': 'chunk_001569',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
