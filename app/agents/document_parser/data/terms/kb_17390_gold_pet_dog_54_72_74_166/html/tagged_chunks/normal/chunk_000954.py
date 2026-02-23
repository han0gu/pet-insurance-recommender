from langchain_core.documents import Document

chunk = Document(
    page_content=("합니다.</p><br><h1 id='143' style='font-size:14px'>예 시 2</h1><br><h1 id='144' "
 "style='font-size:14px'>슬관절/고관절 탈구의 보장개시일</h1><br><table id='145' "
 "style='font-size:20px'><thead></thead><tbody><tr><td>계약일 "
 '보장개시일</td><td></td></tr><tr><td colspan="2"><figure><img alt="1년'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000954',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
