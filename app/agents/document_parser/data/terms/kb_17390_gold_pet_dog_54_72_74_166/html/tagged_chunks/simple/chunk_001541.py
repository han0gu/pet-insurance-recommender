from langchain_core.documents import Document

chunk = Document(
    page_content=(". 손바닥 크기</h1><br><p id='28' data-category='list'></p><br><p id='29' "
 "data-category='paragraph' style='font-size:14px'>‘손바닥 크기’라 함은 해당 환자의 손가락을 "
 '제외한 손바닥의 크기를 말하<br>며, 12세 이상의 성인에서는 8×10㎝(1/2 크기는 40㎠, 1/4 크기는 '
 '20㎠),<br>6~11세의 경우는 6×8㎝(1/2 크기는 24㎠, 1/4 크기는 12㎠), 6세 미만의 경<br>우는 4×6㎝(1/2 '
 '크기는 12㎠, 1/4 크기는'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001541',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
