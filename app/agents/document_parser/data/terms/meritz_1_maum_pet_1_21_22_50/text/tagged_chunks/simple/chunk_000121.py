from langchain_core.documents import Document

chunk = Document(
    page_content=('해 내용증명, 재산조사, 강제집행 등을 수행하고자 지출한 각종 비용을 의미합니\n'
 '다.다. 피보험자가 지급한 소송비용, 변호사비용, 중재, 화해 또는 조정에 관한 비용\n'
 '라. 보험증권상 보상한도액내의 금액에 대한 공탁보증보험료. 그러나 회사는 그러한\n'
 '보증을 제공할 책임은 부담하지 않습니다.\n'
 '마. 피보험자가 제12조(손해배상청구에 대한 회사의 해결)의 제2항 및 제3항의 회사\n'
 '의 요구에 따르기 위하여 지출한 비용【공탁보증보험료】가압류, 가집행, 가처분 신청 등 각종 민사사건을 신청할 때, 잘못된 신청으로'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000121',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
