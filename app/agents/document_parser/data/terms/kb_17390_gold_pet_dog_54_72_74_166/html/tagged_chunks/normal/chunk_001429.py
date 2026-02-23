from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:18px'>- 137 -</p><table id='98' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>계약 보험료는</td><td>보험료 "
 '납입영수증에 장애인전용 보장성보험료로 표시됩니다.</td></tr><tr><td colspan="2">예 시 특별세액공제 대상 기간 예시 '
 '2022년 1월 15일에 전환대상계약에 가입한 계약자가 2022년 6월 1일에 이 특별 약관을 청약하고 회사가 승낙하여 전환대상계약이 '
 '장애인전용보험으로 전환된 경우, 이 특별약관을'),
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
 'indexing': {'chunk_id': 'chunk_001429',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
