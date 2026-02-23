from langchain_core.documents import Document

chunk = Document(
    page_content=("청구를 포기한 경우에도 회사의 제1항에<br>의한 지급보험금 결정에는 영향을 미치지 않습니다.</p><br><p id='193' "
 "data-category='list'></p><br><h1 id='194' style='font-size:14px'>용 어 풀 "
 "이</h1><br><h1 id='195' style='font-size:14px'>공제계약</h1><br><table id='196' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>유사보험으로서 공제</td><td>사업을 "
 '실시하는'),
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
 'indexing': {'chunk_id': 'chunk_001178',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
