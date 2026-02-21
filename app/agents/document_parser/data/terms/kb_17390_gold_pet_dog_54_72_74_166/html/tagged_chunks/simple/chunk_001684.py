from langchain_core.documents import Document

chunk = Document(
    page_content=('풀기, 지퍼 올리고 내리기, 끈 묶고 풀기 등) 이 필요한 마무리는 타인의 도움이 필요한 '
 "상태</td><td>3%</td></tr></tbody></table><p id='24' data-category='paragraph' "
 "style='font-size:14px'>156 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><table "
 "id='25' style='font-size:14px'><thead><tr><td>별표2</td><td>보험금을 지급할 때의 적립이율"),
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
 'indexing': {'chunk_id': 'chunk_001684',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
