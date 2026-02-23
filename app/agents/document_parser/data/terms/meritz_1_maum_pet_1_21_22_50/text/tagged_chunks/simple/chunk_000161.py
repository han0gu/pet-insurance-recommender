from langchain_core.documents import Document

chunk = Document(
    page_content=('수술 당일 발생한 수술비 및 치료비는 보상하여 드리지 않습니다.\n'
 '② 제1항에도 불구하고 보험개시일로부터 그 날을 포함하여 보험증권에 기재된\n'
 '면책(보상하지 않는)기간(이하“ 대기기간”) 이내에 발생한 질병은 보상하지 않습니다.\n'
 '단, 이 계약이 갱신계약인 경우에는 적용하지 않습니다.제2조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.- 31 '
 '-반려묘 비뇨기·전염성복막염 치료비 보장 특별약관제1조(보상하는 손해)① 회사는 보통약관 제5조(보험금을 지급하지 않은 사유) 제2항 '
 '제4호에도 불구하고'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000161',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
