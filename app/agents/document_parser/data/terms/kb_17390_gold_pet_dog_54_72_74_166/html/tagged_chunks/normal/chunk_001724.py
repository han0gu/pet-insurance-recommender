from langchain_core.documents import Document

chunk = Document(
    page_content=("이 약관에서 보장하는 상병 해당 여부를 다시 판단하지 않습니다.</p><p id='54' data-category='paragraph' "
 "style='font-size:18px'>- 158 -</p><table id='55' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>별표7 골절철심제거 수술 "
 '분류표</td></tr><tr><td>약관에 규정하는 "골절철심제거술"은 “건강보험 행위 급여․비급여 목록 및 급여 '
 "상대</td></tr></tbody></table><br><p id='56'"),
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
 'indexing': {'chunk_id': 'chunk_001724',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
