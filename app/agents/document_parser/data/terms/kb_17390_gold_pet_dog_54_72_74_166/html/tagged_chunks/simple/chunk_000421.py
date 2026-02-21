from langchain_core.documents import Document

chunk = Document(
    page_content=("id='95' data-category='paragraph' style='font-size:16px'>구분</p><br><p "
 "id='96' data-category='paragraph' style='font-size:16px'>안면부</p><br><p "
 "id='97' data-category='paragraph' style='font-size:14px'>동<br>상지․하지 "
 "물</p><br><p id='98' data-category='paragraph' style='font-size:14px'>수술 1cm당"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000421',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
