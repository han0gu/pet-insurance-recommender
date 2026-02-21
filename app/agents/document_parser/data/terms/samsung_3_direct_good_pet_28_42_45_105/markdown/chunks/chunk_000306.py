from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 전쟁, 혁명, 내란, 사변, 테러, 폭동, 소요, 노동쟁의, 기타 이들과 유사한 사태\n'
 '- 3. 지진, 분화, 홍수, 해일 또는 이와 비슷한 천재지변\n'
 '- 4. 핵연료물질 또는 핵연료물질에 의하여 오염된 물질의 방사성, 폭발성 또는 그 밖의\n'
 '- 유해한 특성 또는 이들 특성에 의한 사고\n'
 '- 5. 제4호 이외의 방사선을 쬐는 것 또는 방사능 오염\n'
 '<용어풀이># [핵연료물질]사용된 연료를 포함합니다.\n'
 '[핵연료물질에 의하여 오염된 물질]\n'
 '원자핵 분열 생성물을 포함합니다.- 6. 피보험자의 질병, 심신상실 또는 정신질환으로 인한 손해'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
