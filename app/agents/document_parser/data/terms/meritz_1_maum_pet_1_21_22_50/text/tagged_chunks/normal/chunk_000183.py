from langchain_core.documents import Document

chunk = Document(
    page_content=('4. 제1호에도 불구하고 보험의 목적의 정보의 변경에 관한 서류 제출 시기는 계약자와 별\n'
 '도로 협의하여 변경할 수 있습니다.제5조(준용규정)이 추가특별약관에 정하지 않은 사항은 보통약관 및 단체계약 특별약관을 따릅니다.- 39 '
 '-단체계약 보험료정산 추가특별약관(Ⅱ)제1조(보험료의 정산)① 회사는 단체계약 특별약관(이하“특별약관”이라 합니다) 제4조(보험의 목적의 '
 '증가 감소\n'
 '또는 교체) 제2항 및 보통약관 제16조(계약 후 알릴 의무) 제2항에도 불구하고 이 추가\n'
 '특별약관에 따라 보험료를 정산합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000183',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
