from langchain_core.documents import Document

chunk = Document(
    page_content=('- 의 관계는 사고발생 당시의 관계를 말합니다.\n'
 '# 제4조(보험금의 청구)\uf000 보험수익자는 다음의 서류를 제출하고 보험금을 청구하\n'
 '여야 합니다.- ① 청구서(회사 양식)\n'
 '- ② 사고증명서(동물병원 진료비 영수증(진료 항목별 영수\n'
 '- 금액 포함), 동물병원 진료기록부, X-ray 등 방사선\n'
 '- 촬영을 하는 경우 해당 사진(촬영일자 및 시간 필수),\n'
 '91# 기타 지불 증빙서류 등)- ③ 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정\n'
 '- 부기관발행 신분증, 본인이 아닌 경우에는 본인의 인'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000155',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
