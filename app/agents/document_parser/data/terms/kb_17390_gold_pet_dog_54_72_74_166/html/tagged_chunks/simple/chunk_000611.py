from langchain_core.documents import Document

chunk = Document(
    page_content=('어 풀 이 「건강보험 행위</td><td>건강보험심사평가원 진료수가코드(EDI) 급여․비급여 목록 및 급여 상대가치점수(보건복지부 '
 "고시)」</td></tr></tbody></table><br><p id='122' data-category='paragraph' "
 "style='font-size:14px'>에서 정한 처치 및 수술료, 검사료, 방사선 치료료 등을 포함한 항목에 부여되<br>반<br>는 "
 "코드를 말합니다.<br>려<br>동</p><br><p id='123' data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000611',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
