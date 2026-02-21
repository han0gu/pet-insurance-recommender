from langchain_core.documents import Document

chunk = Document(
    page_content=(". 규정<br>※ 향후 관련법령이 개정된 경우 개정된 내용을 적용합니다.</p><p id='26' "
 "data-category='paragraph' style='font-size:18px'>- 55 -</p><br><p id='27' "
 "data-category='paragraph' style='font-size:16px'>KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01) 55</p><br><p id='28' data-category='paragraph' "
 "style='font-size:14px'>공</p><p id='29'"),
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
 'indexing': {'chunk_id': 'chunk_000027',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
