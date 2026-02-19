from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 청구서(회사양식) 2. 등록견의 경우에는 동물등록증 또는 등록번호 3. 미등록견의 경우에는 가입동물의 사진 2매(얼굴전면, '
 '측면전신사진)를 회사에 제출 하시고 가입동물이 보험에 가입한 동물과 동일함을 확인 후 보험금을 지급합니다. 4. 사고증명서(진단서, '
 '진료비계산서, 사망진단서, 장해진단서, 입원치료확인서, 의사 처방전(처방조제비) 등) 5'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 91},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000592',
              'chunk_char_len': 195,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
