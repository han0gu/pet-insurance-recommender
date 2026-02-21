from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 67</p><br><p id='64' "
 "data-category='paragraph' style='font-size:20px'>- 67 -</p><table id='65' "
 "style='font-size:14px'><thead><tr><td></td></tr></thead><tbody><tr><td>용 어 풀 "
 '이 ∙ 강제집행 강제집행이란 사법상 또는 행정법상의 의무를 이행하지 않는 사람에 대하여 국 가가 강제 권력으로 그 의무를 이행하는 것을'),
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
 'indexing': {'chunk_id': 'chunk_000258',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
