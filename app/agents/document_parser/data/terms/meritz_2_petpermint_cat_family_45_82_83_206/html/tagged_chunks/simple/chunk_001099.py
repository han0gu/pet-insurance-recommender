from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보호자나 환자의<br>진술, 감정의의 추정 혹은 인정, 한국표준화가 이<br>루어지지 않고 신빙성이 적은 검사들(뇌 SPECT '
 '등)<br>은 객관적 근거로 인정하지 않는다.<br>타) 각종 기질성 정신장해와 외상후 뇌전증에 한하여<br>보상한다.<br>파) 외상후 '
 '스트레스장애, 우울증(반응성) 등의 질환,<br>정신분열증(조현병), 편집증, 조울증(양극성장<br>애), 불안장애, 전환장애, '
 "공포장애, 강박장애 등<br>각종 신경증 및 각종 인격장애는 보상의 대상이<br>되지 않는다.</p><h1 id='46'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'head', 'urinary']},
 'indexing': {'chunk_id': 'chunk_001099',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
