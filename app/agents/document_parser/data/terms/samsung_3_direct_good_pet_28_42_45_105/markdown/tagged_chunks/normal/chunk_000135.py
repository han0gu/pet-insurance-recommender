from langchain_core.documents import Document

chunk = Document(
    page_content=('- 45 -제2관 보험금의 지급제3조 (보험금의 지급사유)각 특별약관의 보장을 따릅니다.제4조 (보험금 지급에 관한 세부규정)# 각 '
 '특별약관의 보장을 따릅니다.# 제5조 (보험료 납입면제)보험료 납입면제 사항은 기본계약의 보험료 납입면제 사항을 준용합니다.# 제6조 '
 '(보험료 납입면제에 관한 세부규정)보험료 납입면제에 관한 세부규정은 기본계약의 보험료 납입면제에 관한 세부규정을 준\n'
 '용합니다.# 제7조 (보험금을 지급하지 않는 사유)① 회사는 다음 중 어느 한 가지로 각 특별약관별 보험금의 지급사유에서 정한 보험금 지'),
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
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000135',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
