from langchain_core.documents import Document

chunk = Document(
    page_content=("보장됨</p><br><p id='145' data-category='paragraph' style='font-size:14px'>60 "
 "KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><table id='146' "
 "style='font-size:14px'><thead></thead><tbody><tr><td "
 'colspan="2"><table><thead><tr><td>을 통보하고 이에 따라</td><td>보험금을 '
 '지급합니다.</td></tr></thead><tbody><tr><td colspan="2">유 의 사 항'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000113',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
