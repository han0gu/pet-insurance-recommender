from langchain_core.documents import Document

chunk = Document(
    page_content=('지 않습니다.제4조(전환 취소)# 계약자는전환대상계약에 대하여 장애인전용보험으로의 전환을 취소할 수 있으며, 이경우 전환취소 신청서를 '
 '회사에 제출하여야 합니다.제5조(준용규정)\uf000 이특별약관에서 정하지 않은 사항에 대하여는 전환대상계약 약관, 소득세법 등138 '
 'KB 금쪽같은 펫보험(강아지)(무배당)(26.01)# 관련법규에서 정하는 바에 따릅니다.# \uf000 소득세법 등 관련법규가 제·개정 '
 '또는 폐지되는 경우 변경된 법령을 따릅니다.# 7. 반려동물 특정 질병 보장제한부 인수제1조(특별약관의 체결 및 효력)'),
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
 'indexing': {'chunk_id': 'chunk_000818',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
