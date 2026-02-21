from langchain_core.documents import Document

chunk = Document(
    page_content=('- 을 체결할 때 산출한 예치보험료를 비교하여 그 차액을 정산합니다.\n'
 '- 4. 제1호에도 불구하고 보험의 목적의 정보의 변경에 관한 서류 제출 시기는 계약자와 별\n'
 '- 도로 협의하여 변경할 수 있습니다.\n'
 '# 제4조(준용규정)이 추가특별약관에 정하지 않은 사항은 보통약관 및 단체계약 특별약관을 따릅니다.- 40 -# 단체계약 보험기간 설정 '
 '추가특별약관# 제1조(적용범위)이 추가특별약관은 단체계약 특별약관(이하“특별약관”이라 합니다) 제4조(보험의 목적의'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000191',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
