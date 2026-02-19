from langchain_core.documents import Document

chunk = Document(
    page_content=('【설명】\n'
 '제3자로부터 손해의 배상을 받을 수 있는 경우에 피보험자가 손해배상청구를 위 해 내용증명, 재산조사, 강제집행 등을 수행하고자 지출한 '
 '각종 비용을 의미합니 다.\n'
 '다. 피보험자가 지급한 소송비용, 변호사비용, 중재, 화해 또는 조정에 관한 비용 라. 보험증권상 보상한도액내의 금액에 대한 '
 '공탁보증보험료. 그러나 회사는 그러한 보증을 제공할 책임은 부담하지 않습니다. 마. 피보험자가 제12조(손해배상청구에 대한 회사의 '
 '해결)의 제2항 및 제3항의 회사 의 요구에 따르기 위하여 지출한 비용\n'
 '【공탁보증보험료】'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 23},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000142',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
