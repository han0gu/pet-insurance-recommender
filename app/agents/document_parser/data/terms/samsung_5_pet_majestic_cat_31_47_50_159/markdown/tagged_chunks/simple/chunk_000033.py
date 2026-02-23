from langchain_core.documents import Document

chunk = Document(
    page_content=('을 공제한 금액을 말합니다. 이하 같습니다)에 대한 적립이율은 매월 1일 회사가 정한\n'
 '보장성 공시이율Ⅴ(이하「공시이율」이라 합니다)로 하며, 공시이율은 매월 1일부터\n'
 '해당월 마지막 날까지 1개월간 확정 적용합니다.# [공시이율]# <용어풀이>전통적인 보험상품에 적용되는 이율이 장기·고정금리이기 때문에 '
 '시중금리가 급격하게 변동할 경\n'
 '우 이에 대응하지 못하는 점을 고려하여, 시중의 지표금리 등에 연동하여 일정기간 마다 변동되는\n'
 '이율을 말합니다.② 제1항의 공시이율은 이 보험의 사업방법서에서 정한 방법에 따라 회사의 운용자산이'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000033',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
