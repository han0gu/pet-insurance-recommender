from langchain_core.documents import Document

chunk = Document(
    page_content=("장해 평</h1><br><p id='1' data-category='paragraph' style='font-size:16px'>가를 "
 "유보한다.</p><br><p id='2' data-category='paragraph' style='font-size:16px'>마) "
 "장해진단 전문의는 재활의학과, 신경외과 또는 신경과 전문의로 한다.</p><p id='3' data-category='list' "
 "style='font-size:14px'>2) 정신행동<br>가) 정신행동장해는 보험기간중에 발생한 뇌의 질병 또는 상해를 입은<br>후"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_001657',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
