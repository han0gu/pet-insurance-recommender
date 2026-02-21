from langchain_core.documents import Document

chunk = Document(
    page_content=('있는 경우로서 상법 시행령 제44조의2에 정하는 바에 따<br>라 본인 확인 및 위조ㆍ변조 방지에 대한 신뢰성을 갖춘 전자문서를 '
 '포함)으로 동의하<br>여야 합니다.<br>⑥ 회사는 제1항에 따라 계약자를 변경한 경우, 변경된 계약자에게 보험증권 및 약관을 '
 "교<br>부하고 변경된 계약자가 요청하는 경우 약관의 중요한 내용을 설명하여 드립니다.</p><footer id='37' "
 "style='font-size:14px'>- 14 -</footer><p id='38' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000134',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
