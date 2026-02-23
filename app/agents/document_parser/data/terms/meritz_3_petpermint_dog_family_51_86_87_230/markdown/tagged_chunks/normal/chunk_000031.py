from langchain_core.documents import Document

chunk = Document(
    page_content=('된 계약에 대하여 계약자에게 연1회이상 [보장]공시이율의\n'
 '변경내역을 통지합니다.\n'
 '\uf000 세부적인 [보장]공시이율의 운영방법은 회사에서 별도로\n'
 '정한「[보장]공시이율 적용에 관한 세부지침」을 따릅니다.【적립부분 적립이율】적립부분 계약자적립액 계산시 적립부분 순보험료에 대\n'
 '한 이자를 계산할 때 적용하는 이율을 말합니다.58# 【[보장]공시이율】전통적인 보험상품에 적용되는 이율이 장기･고정금리이\n'
 '기 때문에 시중금리가 급격하게 변동할 경우 이에 대응\n'
 '하지 못하는 점을 고려하여, 시중의 지표금리 등에 연동'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000031',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
