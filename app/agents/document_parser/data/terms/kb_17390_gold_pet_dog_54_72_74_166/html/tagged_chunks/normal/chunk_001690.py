from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사가 해지권을 행사하는 경우 위 표의 청구일은 회사의 해지 의사표시(서 면, 전자우편, 휴대전화 문자메시지 또는 이에 준하는 전자적 '
 "의사표시 포 함)가 보험계약자 또는 그의 대리인에게 도달한 날로 봅니다.</td></tr></tbody></table><p id='27' "
 "data-category='paragraph' style='font-size:18px'>- 156 -</p><p id='28' "
 "data-category='paragraph' style='font-size:16px'>별표3 외모특정상해 분류표<br>\uf000 "
 '약관에'),
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
 'indexing': {'chunk_id': 'chunk_001690',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
