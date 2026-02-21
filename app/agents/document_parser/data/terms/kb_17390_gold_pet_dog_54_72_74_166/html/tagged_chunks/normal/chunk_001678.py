from langchain_core.documents import Document

chunk = Document(
    page_content=('사용이 불가능하여 음 식물 섭취에 있어 부분적으로 다른 사람의 도움이 필 10% 요한 '
 '상태</td><td>규정</td></tr><tr><td>4) 독립적인 음식물 섭취는 가능하나 젓가락을 이용하여 5% 생선을 바르거나 '
 "음식물을 자르지는 못하는 상태</td><td></td></tr></tbody></table><p id='21' "
 "data-category='paragraph' style='font-size:14px'>KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01) 155</p><br><p id='22'"),
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
 'indexing': {'chunk_id': 'chunk_001678',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
