from langchain_core.documents import Document

chunk = Document(
    page_content=('| 58 | 연골증 | M94 | 연골의 기타 장애 |\n'
 '주) 제10차 개정 이후 한국표준질병·사인분류에서 상기 분류표에 변경사항이 발생하는 경우에는 변\n'
 '경된 분류표에 따릅니다.- 103 -103 / 1814-5. 특별조건부 특별약관# 제 1조 (보험계약의 성립)- ① 이 특별약관은 '
 '보험계약(특별약관이 부가된 경우에는 그 특별약관을 포함합니다. 이하\n'
 '- 「보험계약」이라 합니다)을 체결할 때 피보험자의 건강상태가 보험회사(이하「회\n'
 '- 사」라 합니다)가 정한 기준에 적합하지 않은 경우 보험계약자(이하「계약자」라 합니'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000580',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
