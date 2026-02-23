from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 동물병원에서 수의사에게<br>수술을 받은 경우 수술 당일 발생한 수술비 및 치료비는 보상하여 드리지 않습니다.<br>② '
 '제1항에도 불구하고 보험개시일로부터 그 날을 포함하여 보험증권에 기재된<br>면책(보상하지 않는)기간(이하“ 대기기간”) 이내에 발생한 '
 "질병은 보상하지 않습니다.<br>단, 이 계약이 갱신계약인 경우에는 적용하지 않습니다.</p><p id='9' "
 "data-category='paragraph' style='font-size:14px'>제2조(준용규정)</p><br><p id='10'"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000290',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
