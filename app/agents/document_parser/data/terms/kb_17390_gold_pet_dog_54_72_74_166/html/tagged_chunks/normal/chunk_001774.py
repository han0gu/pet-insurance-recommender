from langchain_core.documents import Document

chunk = Document(
    page_content=("보장하는 질병 해당 여부를 다시 판단하지 않습니다.</p><table id='93' "
 "style='font-size:14px'><thead></thead><tbody><tr><td></td><td>별표13 천식지속상태 "
 "분류표</td></tr></tbody></table><br><p id='94' data-category='paragraph' "
 "style='font-size:14px'>\uf000 약관에 규정하는 천식지속상태로 분류되는 질병은 제9차 개정 "
 "한국표준질병․사인</p><br><p id='95'"),
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
 'indexing': {'chunk_id': 'chunk_001774',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
