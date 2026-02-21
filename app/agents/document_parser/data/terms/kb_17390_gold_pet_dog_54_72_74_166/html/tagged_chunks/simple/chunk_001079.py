from langchain_core.documents import Document

chunk = Document(
    page_content=('2024년 4월 10일 2024년 5월 9일" data-coord="top-left:(155,416); '
 'bottom-right:(736,524)" /></figure></td></tr></tbody></table><br><p '
 "id='56' data-category='paragraph' style='font-size:16px'>\uf000 제3항에도 불구하고 "
 '"제도성 특별약관 - 보장특약 자동갱신(추가납입형) 특별약<br>관"에서 정한 방법에 따라 갱신된 계약의 무지개다리위로금보장개시일은 이 '
 '특별<br>약관의 갱신일로 합니다.</p><p'),
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
 'indexing': {'chunk_id': 'chunk_001079',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
