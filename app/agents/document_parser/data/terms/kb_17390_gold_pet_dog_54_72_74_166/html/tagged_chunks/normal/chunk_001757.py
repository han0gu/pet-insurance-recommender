from langchain_core.documents import Document

chunk = Document(
    page_content=('창상봉합술을 시행할경우 소 정점수에 103.14점을 가산하며, 창상봉합 길 이가 10cm 증가될때마다 103.14점을 추가 가 '
 "산한다.</td><td>SC040</td></tr></tbody></table><br><p id='77' "
 "data-category='paragraph' style='font-size:16px'>별표11 2대호흡계특정질환 "
 '분류표<br>공<br>\uf000 약관에 규정하는 2대호흡계특정질환으로 분류되는 질병은 제9차 개정 한국표준질<br>통<br>병 '
 '․사인분류(KCD, 통계청 고시 제2025-299호, 2026.1.1'),
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
 'indexing': {'chunk_id': 'chunk_001757',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
