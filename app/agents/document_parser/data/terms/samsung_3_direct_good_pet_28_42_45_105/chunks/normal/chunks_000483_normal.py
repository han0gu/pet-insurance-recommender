from langchain_core.documents import Document

chunk = Document(
    page_content=('5. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발생 신분증, 본인이 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 '
 '또는 안전성과 신뢰성이 확보된 전자적 수단을 활용한 보험수익자 의사표시의 확인방법 포함) 6. 기타 보험수익자가 보험금의 수령에 필요하여 '
 '제출하는 서류\n'
 '② 제1항 제4호의 사고증명서는 수의사법 제2조(정의)에서 규정한 동물병원에서 수의사 가 발급한 것이어야 합니다.\n'
 '<관련법규>\n'
 '[수의사법 제2조(정의)]'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 79},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000483',
              'chunk_char_len': 246,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
